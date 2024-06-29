from collections import deque
from code_parser.p_ast import P_AST, parsers

class JAVA_AST(P_AST):
    def __init__(self, root, code, idx=0, parent=None, deep=0):
        super().__init__(root, code, JAVA_AST, idx, parent, deep)

    @classmethod
    def build_ast(cls, code, lang="java"):
        the_code = bytes(code, 'utf8')
        node = parsers[lang].parse(the_code)
        the_ast = JAVA_AST(node.root_node, the_code)
        the_ast.link_ast()
        return the_ast

    @staticmethod
    def check_is_function_name(node):
        if node.path.endswith("method_declaration|identifier"):
            return True
        if node.path.endswith("constructor_declaration|identifier"):
            return True
        return False

    @staticmethod
    def check_is_function(node):
        if node.path.endswith("method_declaration"):
            return True
        if node.path.endswith("constructor_declaration"):
            return True
        return False

    @staticmethod
    def check_is_nest_function(node):
        functions = node.get_functions()
        return False if len(functions) == 1 else True
    
    @staticmethod
    def check_is_constructor(node):
        if node.path.endswith("constructor_declaration"):
            return True
        return False
    
    @staticmethod
    def check_is_comment(node):
        if node.path.endswith("comment"):
            return True
        return False
    
    def get_class_source(self):
        result = self.bfs_search_one_source(self, lambda node: node.path.endswith("program|class_declaration"))
        if result is not None:
            return result.source
        return None
    
    def get_class_node(self):
        return self.bfs_search_one(self, lambda node: node.path.endswith("program|class_declaration"))
    
    def get_class_name(self):
        result = self.bfs_search_one_source(self, 
                                     lambda node: node.path.endswith("program|class_declaration|identifier"),
                                     assert_check=lambda node: node.path.endswith("program|class_declaration"))
        if result is not None:
            return result.source
        return None
      
    def get_function_body(self):
        return self.bfs_search_one(self, 
                                   lambda node: node.path.endswith("method_declaration|block") or node.path.endswith("constructor_declaration|constructor_body"),
                                   assert_check=lambda node: node.check_is_function(node))

    def get_package_source(self):
        return self.bfs_search_one_source(self, 
                                         lambda node: node.path.endswith("program|package_declaration"))
    
    def get_imports_source(self):
        return self.bfs_search_all_source(self, 
                                         lambda node: node.path.endswith("program|import_declaration"))
    
    def get_class_signature_source(self):
        assert self.path.endswith("program|class_declaration"), "Not a class"
        class_signature = ""
        for child in self.children:
            if child.path.endswith("program|class_declaration|class_body"): continue
            class_signature += child.source
        return class_signature
    
    def get_field_source(self):
        return self.bfs_search_all_source(self, 
                                         lambda node: node.path.endswith("program|class_declaration|class_body|field_declaration"))
    
    def get_class_functions(self):
        return self.bfs_search_all(self, 
                                   lambda node: node.path.endswith("class_body|constructor_declaration") 
                                   or node.path.endswith("class_body|method_declaration"))
    
    def get_function_signature_source(self):
        assert self.check_is_function(self), "Not a function"
        function_signature = ""
        for child in self.children:
            if child.path.endswith("method_declaration|block"): continue
            if child.path.endswith("constructor_declaration|constructor_body"): continue
            function_signature += child.source
        return function_signature
    
    def get_class_functions_signature_source(self):
        return [func.get_function_signature_source() for func in self.get_class_functions()]
    
    def get_fill_in(self):
        return self.bfs_search_all_source(self, 
                                   lambda node: node.path.endswith("fill_in"))
    
    def get_indent(self):
        fill_in = self.get_fill_in()
        line_fill_in = [fill.replace("\r\n", "").replace("\n", "").replace("\t", "    ") for fill in fill_in if "\n" in fill]
        line_fill_in = [fill for fill in line_fill_in if fill != ""]
        line_fill_in.sort(key=lambda x: len(x))
        return line_fill_in[0]
    
    def get_import_context_source(self):
        assert self.path.endswith("program"), "Not a file"
        context = ""
        
        package_source = self.get_package_source()
        if package_source: context += package_source + "\n\n"
        
        import_source = "\n".join(self.get_imports_source())
        if import_source: context += import_source + "\n\n"
        
        return context
    
    def get_import_context_source(self):
        assert self.path.endswith("program"), "Not a file"
        context = ""
        
        package_source = self.get_package_source()
        if package_source: context += package_source + "\n\n"
        
        import_source = "\n".join(self.get_imports_source())
        if import_source: context += import_source + "\n\n"
        
        return context
      
    def get_file_context_source(self):
        assert self.path.endswith("program"), "Not a file"
        indent = self.get_indent()
        context = ""
        
        package_source = self.get_package_source()
        if package_source: context += package_source + "\n\n"
        
        import_source = "\n".join(self.get_imports_source())
        if import_source: context += import_source + "\n\n"
        
        context += self.get_class_context_source()
        
        return context
    
    def get_class_signature_context_source(self):
        assert self.path.endswith("program"), "Not a file"
        
        class_node = self.get_class_node()
        if class_node is None: raise Exception("No class")
        context = class_node.get_class_signature_source()
        
        return context
    
    def get_class_field_context_source(self):
        assert self.path.endswith("program"), "Not a file"
        indent = self.get_indent()
        context = ""
        
        field_source = "\n".join([indent+field for field in self.get_field_source()])
        if field_source: context += field_source
        
        return context
    
    def get_class_functions_signature_context_source(self):
        assert self.path.endswith("program"), "Not a file"
        indent = self.get_indent()
        context = ""
        
        functino_source = []
        for func in self.get_class_functions_signature_source():
            func = func.strip()
            if not func.endswith(";"):
                func += ";"
            functino_source.append(func)
            
        functino_source = "\n".join([indent+func for func in functino_source])
        if functino_source: context += functino_source
        
        return context
    
    def get_class_context_source(self):
        assert self.path.endswith("program"), "Not a file"
        indent = self.get_indent()
        context = ""
        
        class_node = self.get_class_node()
        if class_node is None: raise Exception("No class")
        class_signature_source = class_node.get_class_signature_source()
        if class_signature_source: context += class_signature_source + "{\n\n"
        
        field_source = "\n".join([indent+field for field in self.get_field_source()])
        if field_source: context += field_source + "\n\n"
        
        functino_source = []
        for func in self.get_class_functions_signature_source():
            func = func.strip()
            if not func.endswith(";"):
                func += ";"
            functino_source.append(func)
            
        functino_source = "\n".join([indent+func for func in functino_source])
        if functino_source: context += functino_source + "\n\n"
        
        context += "}\n"
        
        return context
    
if __name__ == "__main__":
    code = """
package com.example;

import java.util.List;
import java.util.ArrayList;

public class Example {
    String name = "a";
    static String a = "a";
    
    public Example() {
        System.out.println("hello world");
    }
    
    //参数是生成的xml文件的路径与名字
    public static void main(String[] args) {
        List<String> list = new ArrayList<>();
        list.add("hello");
        list.add("world");
        for (String s : list) {
            System.out.println(s);
        }
    }
    
    /*
    （1）遍历positiveWord这个Map，得到里面的各个词语在积极词汇中的次数，再在其他两个Map中查看是否有这个词语，有，就把其他的那个
        次数加到当前Map的当前词语的value上，并且删除那个Map中的当前词语；没有这个词的话，那么在那个；类别中出现的次数就是0.
    （2）遍历negativeWord，不用看positiveWord了，只需看unsureWord，处理方法同上。
    （3）遍历unsureWord，这些词在其他两个类别中都是0，直接得到在当前类别中的值
    */
    public static void main(String[] args) {
        List<String> list = new ArrayList<>();
        list.add("hello");
        list.add("world");
        for (String s : list) {
            System.out.println(s);
        }
    }
}
    """
    
    code2 = """
/*
 * Copyright (C) 2011 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.gson.internal.bind;

import com.google.gson.FieldNamingStrategy;
import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;
import com.google.gson.TypeAdapter;
import com.google.gson.TypeAdapterFactory;
import com.google.gson.annotations.JsonAdapter;
import com.google.gson.annotations.SerializedName;
import com.google.gson.internal.$Gson$Types;
import com.google.gson.internal.ConstructorConstructor;
import com.google.gson.internal.Excluder;
import com.google.gson.internal.ObjectConstructor;
import com.google.gson.internal.Primitives;
import com.google.gson.reflect.TypeToken;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonToken;
import com.google.gson.stream.JsonWriter;

import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static com.google.gson.internal.bind.JsonAdapterAnnotationTypeAdapterFactory.getTypeAdapter;

/**
 * Type adapter that reflects over the fields and methods of a class.
 */
public final class ReflectiveTypeAdapterFactory implements TypeAdapterFactory {
  private final ConstructorConstructor constructorConstructor;
  private final FieldNamingStrategy fieldNamingPolicy;
  private final Excluder excluder;

  public ReflectiveTypeAdapterFactory(ConstructorConstructor constructorConstructor,
      FieldNamingStrategy fieldNamingPolicy, Excluder excluder) {
    this.constructorConstructor = constructorConstructor;
    this.fieldNamingPolicy = fieldNamingPolicy;
    this.excluder = excluder;
  }

  public boolean excludeField(Field f, boolean serialize) {
    return excludeField(f, serialize, excluder);
  }

  static boolean excludeField(Field f, boolean serialize, Excluder excluder) {
    return !excluder.excludeClass(f.getType(), serialize) && !excluder.excludeField(f, serialize);
  }

  /** first element holds the default name */
  private List<String> getFieldNames(Field f) {
    SerializedName annotation = f.getAnnotation(SerializedName.class);
    if (annotation == null) {
      String name = fieldNamingPolicy.translateName(f);
      return Collections.singletonList(name);
    }

    String serializedName = annotation.value();
    String[] alternates = annotation.alternate();
    if (alternates.length == 0) {
      return Collections.singletonList(serializedName);
    }

    List<String> fieldNames = new ArrayList<String>(alternates.length + 1);
    fieldNames.add(serializedName);
    for (String alternate : alternates) {
      fieldNames.add(alternate);
    }
    return fieldNames;
  }

  @Override public <T> TypeAdapter<T> create(Gson gson, final TypeToken<T> type) {
    Class<? super T> raw = type.getRawType();

    if (!Object.class.isAssignableFrom(raw)) {
      return null; // it's a primitive!
    }

    ObjectConstructor<T> constructor = constructorConstructor.get(type);
    return new Adapter<T>(constructor, getBoundFields(gson, type, raw));
  }

  private ReflectiveTypeAdapterFactory.BoundField createBoundField(
      final Gson context, final Field field, final String name,
      final TypeToken<?> fieldType, boolean serialize, boolean deserialize) {
    final boolean isPrimitive = Primitives.isPrimitive(fieldType.getRawType());
    // special casing primitives here saves ~5% on Android...
    JsonAdapter annotation = field.getAnnotation(JsonAdapter.class);
    TypeAdapter<?> mapped = null;
    if (annotation != null) {
      mapped = getTypeAdapter(constructorConstructor, context, fieldType, annotation);
    }
    final boolean jsonAdapterPresent = mapped != null;
    if (mapped == null) mapped = context.getAdapter(fieldType);

    final TypeAdapter<?> typeAdapter = mapped;
    return new ReflectiveTypeAdapterFactory.BoundField(name, serialize, deserialize) {
      @SuppressWarnings({"unchecked", "rawtypes"}) // the type adapter and field type always agree
      @Override void write(JsonWriter writer, Object value)
          throws IOException, IllegalAccessException {
        Object fieldValue = field.get(value);
        TypeAdapter t = jsonAdapterPresent ? typeAdapter
            : new TypeAdapterRuntimeTypeWrapper(context, typeAdapter, fieldType.getType());
        t.write(writer, fieldValue);
      }
      @Override void read(JsonReader reader, Object value)
          throws IOException, IllegalAccessException {
        Object fieldValue = typeAdapter.read(reader);
        if (fieldValue != null || !isPrimitive) {
          field.set(value, fieldValue);
        }
      }
      @Override public boolean writeField(Object value) throws IOException, IllegalAccessException {
        if (!serialized) return false;
        Object fieldValue = field.get(value);
        return fieldValue != value; // avoid recursion for example for Throwable.cause
      }
    };
  }

  private Map<String, BoundField> getBoundFields(Gson context, TypeToken<?> type, Class<?> raw) {
    Map<String, BoundField> result = new LinkedHashMap<String, BoundField>();
    if (raw.isInterface()) {
      return result;
    }

    Type declaredType = type.getType();
    while (raw != Object.class) {
      Field[] fields = raw.getDeclaredFields();
      for (Field field : fields) {
        boolean serialize = excludeField(field, true);
        boolean deserialize = excludeField(field, false);
        if (!serialize && !deserialize) {
          continue;
        }
        field.setAccessible(true);
        Type fieldType = $Gson$Types.resolve(type.getType(), raw, field.getGenericType());
        List<String> fieldNames = getFieldNames(field);
        BoundField previous = null;
        for (int i = 0; i < fieldNames.size(); ++i) {
          String name = fieldNames.get(i);
          if (i != 0) serialize = false; // only serialize the default name
          BoundField boundField = createBoundField(context, field, name,
              TypeToken.get(fieldType), serialize, deserialize);
          BoundField replaced = result.put(name, boundField);
          if (previous == null) previous = replaced;
        }
        if (previous != null) {
          throw new IllegalArgumentException(declaredType
              + " declares multiple JSON fields named " + previous.name);
        }
      }
      type = TypeToken.get($Gson$Types.resolve(type.getType(), raw, raw.getGenericSuperclass()));
      raw = type.getRawType();
    }
    return result;
  }

  static abstract class BoundField {
    final String name;
    final boolean serialized;
    final boolean deserialized;

    protected BoundField(String name, boolean serialized, boolean deserialized) {
      this.name = name;
      this.serialized = serialized;
      this.deserialized = deserialized;
    }
    abstract boolean writeField(Object value) throws IOException, IllegalAccessException;
    abstract void write(JsonWriter writer, Object value) throws IOException, IllegalAccessException;
    abstract void read(JsonReader reader, Object value) throws IOException, IllegalAccessException;
  }

  public static final class Adapter<T> extends TypeAdapter<T> {
    private final ObjectConstructor<T> constructor;
    private final Map<String, BoundField> boundFields;

    Adapter(ObjectConstructor<T> constructor, Map<String, BoundField> boundFields) {
      this.constructor = constructor;
      this.boundFields = boundFields;
    }

    @Override public T read(JsonReader in) throws IOException {
      if (in.peek() == JsonToken.NULL) {
        in.nextNull();
        return null;
      }

      T instance = constructor.construct();

      try {
        in.beginObject();
        while (in.hasNext()) {
          String name = in.nextName();
          BoundField field = boundFields.get(name);
          if (field == null || !field.deserialized) {
            in.skipValue();
          } else {
            field.read(in, instance);
          }
        }
      } catch (IllegalStateException e) {
        throw new JsonSyntaxException(e);
      } catch (IllegalAccessException e) {
        throw new AssertionError(e);
      }
      in.endObject();
      return instance;
    }

    @Override public void write(JsonWriter out, T value) throws IOException {
      if (value == null) {
        out.nullValue();
        return;
      }

      out.beginObject();
      try {
        for (BoundField boundField : boundFields.values()) {
          if (boundField.writeField(value)) {
            out.name(boundField.name);
            boundField.write(out, value);
          }
        }
      } catch (IllegalAccessException e) {
        throw new AssertionError(e);
      }
      out.endObject();
    }
  }
}
    """

    the_ast = JAVA_AST.build_ast(code2, lang="java")
    the_ast.print_path_ast()
    
    # print(the_ast.get_package_source())
    # print(the_ast.get_import_source())
    # print(the_ast.get_class_node().get_class_signature_source())
    # print(the_ast.get_field_source())
    # print(the_ast.get_class_functions_signature_source())
    
    # print(the_ast.get_fill_in())
    # print(the_ast.get_indent())
    print(the_ast.get_file_context_source())
    print(the_ast.get_class_context_source())
    

    # functions = the_ast.get_functions()
    # for function in functions:
    #     # if function.get_function_name() is None:
    #     print(function.get_function_name(), function.start_line, function.end_line)
    #     print(function.source)
    #     print(function.source_line)
    #     print(function.get_function_body().source if function.get_function_body() is not None else None)
        
